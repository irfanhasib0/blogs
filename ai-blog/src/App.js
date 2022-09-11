import logo from './logo.svg';
import './App.css';

import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';
import 'bootstrap/dist/css/bootstrap.min.css';

import { Sidebar, Menu, MenuItem, SubMenu, SidebarContent, SidebarHeader, SidebarFooter } from './components/sidebar';
import bgImage from './components/sidebar/assets/bg1.jpg'
import './components/sidebar/scss/styles.scss';
import { FaTachometerAlt, FaRegEdit, FaTrashAlt, FaGem, FaList, FaRegLaughWink, FaHeart, FaBook, FaUserCircle } from 'react-icons/fa';
import {BsJournalRichtext, BsBook, BsStack} from 'react-icons/bs';
import {AiOutlineMail, AiOutlineLogin, AiOutlineLogout} from 'react-icons/ai';


//<Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={props.callbacks.renameTopic} ><FaRegEdit/></Button>
//<Button variant='outline-danger' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={props.callbacks.deleteTopic} ><FaTrashAlt/></Button>
function Header(props){
        return (
	<Navbar bg='gray'>
	<Container>
		<Navbar.Brand href="#home"> <BsBook/> {'Header ...'} 
               </Navbar.Brand>
		<Navbar.Toggle aria-controls="basic-navbar-nav" />
		<Navbar.Collapse id="basic-navbar-nav">
		  <Nav className="me-auto">
		    <Nav.Link href="#home"><FaUserCircle/> {''}</Nav.Link>
		    <NavDropdown title= "Editor" id="basic-nav-dropdown">
		      <NavDropdown.Item href="#action/3.1">Action</NavDropdown.Item>
		      <NavDropdown.Item href="#action/3.2">
		        Another action
		      </NavDropdown.Item>
		      <NavDropdown.Item href="#action/3.3">Something</NavDropdown.Item>
		      <NavDropdown.Divider />
		      <NavDropdown.Item href="#action/3.4">
		        Separated link
		      </NavDropdown.Item>
		    </NavDropdown>
		  </Nav>
		</Navbar.Collapse>
	  </Container>
	  </Navbar>)
 }
 
 
 
function App() {
  //return <><a href='/abc.html'>'chapter-1'</a></>
  //style = {{marginTop : '10px' , 'min-height' : screen.height}}
  return (
  <>
  <Row>
  <Col>
  <Sidebar image={bgImage}> 
  <SidebarHeader style = {{marginTop : '10px', marginLeft : '10px'}}>
  </SidebarHeader>
  <SidebarContent style = {{marginTop : '10px' , minHeight : 1000}}> 
     <Menu iconShape="circle">
     <SubMenu defaultOpen={true} title={'Notebooks'} icon={<BsStack />} suffix={<span className="badge red">{''}</span>}>
     </SubMenu>
     </Menu>
  </SidebarContent>
  <SidebarFooter>
  <div style={{'height':'50px'}}>
  <AiOutlineMail/> {'irfanhasib.me@gmail.com'}
  </div>
  </SidebarFooter>
  </Sidebar>
  </Col>
  <Col xs={9} style={{marginLeft : '0px'}}>
  <Header/>
  <iframe src={"/rcnn_1.svg"} title={"description"} width="100%" height="800px"></iframe>
  {''}
  <footer style={{'position':  'absolute', 'height' : '60px'}}>
  <AiOutlineMail/> {'irfanhasib.me@gmail.com'}
  </footer>
  </Col>
  </Row>
  </>)
}

export default App;
